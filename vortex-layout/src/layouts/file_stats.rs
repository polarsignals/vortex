// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::marker::PhantomData;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::layouts::zoned::MAX_IS_TRUNCATED;
use crate::layouts::zoned::MIN_IS_TRUNCATED;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

pub fn accumulate_stats(
    stream: SendableSequentialStream,
    stats: Arc<[AggregateFnRef]>,
    max_variable_length_statistics_size: usize,
    session: &VortexSession,
    write_legacy_stats: bool,
) -> (FileStatsAccumulator, SendableSequentialStream) {
    let accumulator = FileStatsAccumulator::new(
        stream.dtype(),
        stats,
        max_variable_length_statistics_size,
        session,
        write_legacy_stats,
    );
    let stream = SequentialStreamAdapter::new(
        stream.dtype().clone(),
        stream.scan(accumulator.clone(), |acc, item| {
            future::ready(Some(acc.process(item)))
        }),
    )
    .sendable();
    (accumulator, stream)
}

/// Accumulates write-time statistics for a single file column.
struct StatsAccumulator {
    aggregates: Vec<(AggregateFnRef, AccumulatorRef)>,
    /// Bespoke truncating builders for Utf8/Binary Min/Max, which track per-value exactness
    /// (`Precision::Exact` vs `Inexact`) that the generic `BoundedMax`/`BoundedMin` aggregate fns
    /// can't express (they treat truncatability as a static, config-level property, not a
    /// per-value fact).
    truncated: Vec<Box<dyn StatsArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    fn new(
        dtype: &DType,
        stats: &[AggregateFnRef],
        max_variable_length_statistics_size: usize,
    ) -> Self {
        if !supports_file_stats(dtype) {
            return Self {
                aggregates: Vec::new(),
                truncated: Vec::new(),
                length: 0,
            };
        }

        let is_varlen = is_varlen_dtype(dtype);
        let mut aggregates = Vec::new();
        let mut truncated: Vec<Box<dyn StatsArrayBuilder>> = Vec::new();

        for aggregate_fn in stats {
            if is_varlen && (aggregate_fn.is::<Min>() || aggregate_fn.is::<Max>()) {
                if let Some(stat_dtype) = aggregate_fn.return_dtype(dtype) {
                    truncated.push(stats_builder_with_capacity(
                        aggregate_fn.clone(),
                        &stat_dtype.as_nullable(),
                        1024,
                        max_variable_length_statistics_size,
                    ));
                }
                continue;
            }

            // A dtype that doesn't support a given aggregate simply fails to build an
            // accumulator, which is silently skipped, matching this stat's absence from the
            // result.
            if let Ok(accumulator) = aggregate_fn.accumulator(dtype) {
                aggregates.push((aggregate_fn.clone(), accumulator));
            }
        }

        Self {
            aggregates,
            truncated,
            length: 0,
        }
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        for (_, accumulator) in &mut self.aggregates {
            accumulator.accumulate(array, ctx)?;
        }
        for builder in &mut self.truncated {
            let stat = Stat::from_aggregate_fn(&builder.stat())
                .vortex_expect("truncated builder stat is always min or max");
            if let Some(value) = array.statistics().compute_stat(stat, ctx)? {
                builder.append_scalar(value.cast(&value.dtype().as_nullable())?)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    /// Builds the intermediate per-chunk table backing the `truncated` (Utf8/Binary Min/Max)
    /// builders, from which the file-wide truncated aggregate is re-derived. Returns `None` if
    /// there are no such builders, or all their columns ended up all-null.
    fn truncated_array(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<Option<StructArray>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();

        for builder in self
            .truncated
            .iter_mut()
            // We sort the stats so the DType is deterministic based on which stats are present.
            // `AggregateFnRef` has no `Ord`, so use a proxy key that reproduces the same
            // Max-before-Min order the legacy `Stat` discriminants gave us.
            .sorted_unstable_by_key(|builder| !builder.stat().is::<Max>())
        {
            let values = builder.finish();

            // We drop any all-null stats columns.
            if values.all_invalid(ctx)? {
                continue;
            }

            names.extend(values.names);
            fields.extend(values.arrays);
        }

        if names.is_empty() {
            return Ok(None);
        }

        StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable).map(Some)
    }

    /// Returns an aggregated stats set for the table.
    fn as_stats_set(&mut self, ctx: &mut ExecutionCtx) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();

        for (aggregate_fn, accumulator) in &self.aggregates {
            if let Some(stat) = Stat::from_aggregate_fn(aggregate_fn)
                && let Some(v) = accumulator.final_scalar()?.into_value()
            {
                stats_set.set(stat, Precision::exact(v));
            }
        }

        let Some(stats_table) = self.truncated_array(ctx)? else {
            return Ok(stats_set);
        };

        for builder in &self.truncated {
            let aggregate_fn = builder.stat();
            let Some(stat) = Stat::from_aggregate_fn(&aggregate_fn) else {
                continue;
            };
            let Some(values) = stats_table.unmasked_field_by_name_opt(stat.name()) else {
                continue;
            };

            if stat == Stat::Max && !values.all_valid(ctx)? {
                // A null truncated varlen max can mean either an empty chunk or no finite
                // upper bound, so aggregating by skipping nulls would be unsound.
                continue;
            }

            if let Some(s) = values.statistics().compute_stat(stat, ctx)?
                && let Some(v) = s.into_value()
            {
                let precision = if stat_was_truncated(&stats_table, aggregate_fn, ctx)? {
                    Precision::inexact(v)
                } else {
                    Precision::exact(v)
                };
                stats_set.set(stat, precision);
            }
        }

        Ok(stats_set)
    }
}

fn stat_was_truncated(
    stats_table: &StructArray,
    aggregate_fn: AggregateFnRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let field_name = if aggregate_fn.is::<Min>() {
        MIN_IS_TRUNCATED
    } else if aggregate_fn.is::<Max>() {
        MAX_IS_TRUNCATED
    } else {
        return Ok(false);
    };
    let Some(is_truncated) = stats_table.unmasked_field_by_name_opt(field_name) else {
        return Ok(false);
    };

    Ok(is_truncated
        .statistics()
        .compute_stat(Stat::Max, ctx)?
        .is_some_and(|max| max.as_bool().value() == Some(true)))
}

fn supports_file_stats(dtype: &DType) -> bool {
    !matches!(dtype, DType::Variant(_))
}

fn is_varlen_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Utf8(_) | DType::Binary(_))
}

/// Builds a bespoke truncating builder for Utf8/Binary Min/Max.
///
/// # Panics
///
/// Panics if `aggregate_fn` is not `Min` or `Max`, or `dtype` is not `Utf8`/`Binary` — callers
/// only reach this for varlen Min/Max (see [`StatsAccumulator::new`]).
fn stats_builder_with_capacity(
    aggregate_fn: AggregateFnRef,
    dtype: &DType,
    capacity: usize,
    max_length: usize,
) -> Box<dyn StatsArrayBuilder> {
    let values_builder = builder_with_capacity_in(
        dtype,
        capacity,
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );
    let is_max = aggregate_fn.is::<Max>();
    match dtype {
        DType::Utf8(_) if is_max => Box::new(TruncatedMaxBinaryStatsBuilder::<BufferString>::new(
            values_builder,
            BoolBuilder::with_capacity_in(
                Nullability::NonNullable,
                capacity,
                vortex_buffer::BufferAllocatorRef::static_ref(),
            ),
            max_length,
        )),
        DType::Binary(_) if is_max => Box::new(TruncatedMaxBinaryStatsBuilder::<ByteBuffer>::new(
            values_builder,
            BoolBuilder::with_capacity_in(
                Nullability::NonNullable,
                capacity,
                vortex_buffer::BufferAllocatorRef::static_ref(),
            ),
            max_length,
        )),
        DType::Utf8(_) => Box::new(TruncatedMinBinaryStatsBuilder::<BufferString>::new(
            values_builder,
            BoolBuilder::with_capacity_in(
                Nullability::NonNullable,
                capacity,
                vortex_buffer::BufferAllocatorRef::static_ref(),
            ),
            max_length,
        )),
        DType::Binary(_) => Box::new(TruncatedMinBinaryStatsBuilder::<ByteBuffer>::new(
            values_builder,
            BoolBuilder::with_capacity_in(
                Nullability::NonNullable,
                capacity,
                vortex_buffer::BufferAllocatorRef::static_ref(),
            ),
            max_length,
        )),
        _ => unreachable!("stats_builder_with_capacity is only called for varlen Min/Max"),
    }
}

/// Arrays with their associated names, reduced version of a `StructArray`.
struct NamedArrays {
    names: Vec<FieldName>,
    arrays: Vec<ArrayRef>,
}

impl NamedArrays {
    fn all_invalid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.arrays[0].all_invalid(ctx)
    }
}

trait StatsArrayBuilder: Send {
    fn stat(&self) -> AggregateFnRef;

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()>;

    fn append_null(&mut self);

    fn finish(&mut self) -> NamedArrays;
}

struct TruncatedMaxBinaryStatsBuilder<T: ScalarTruncation> {
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    max_value_length: usize,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedMaxBinaryStatsBuilder<T> {
    fn new(
        values: Box<dyn ArrayBuilder>,
        is_truncated: BoolBuilder,
        max_value_length: usize,
    ) -> Self {
        Self {
            values,
            is_truncated,
            max_value_length,
            _marker: PhantomData,
        }
    }
}

struct TruncatedMinBinaryStatsBuilder<T: ScalarTruncation> {
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    max_value_length: usize,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedMinBinaryStatsBuilder<T> {
    fn new(
        values: Box<dyn ArrayBuilder>,
        is_truncated: BoolBuilder,
        max_value_length: usize,
    ) -> Self {
        Self {
            values,
            is_truncated,
            max_value_length,
            _marker: PhantomData,
        }
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMaxBinaryStatsBuilder<T> {
    fn stat(&self) -> AggregateFnRef {
        Stat::Max
            .aggregate_fn()
            .vortex_expect("max has an aggregate fn")
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        if let Some((upper_bound, truncated)) =
            upper_bound(T::from_scalar(value)?, self.max_value_length, nullability)
        {
            self.values.append_scalar(&upper_bound)?;
            self.is_truncated.append_value(truncated);
        } else {
            self.append_null()
        }
        Ok(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self.values.as_mut());
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![Stat::Max.name().into(), MAX_IS_TRUNCATED.into()],
            arrays: vec![
                ArrayBuilder::finish(self.values.as_mut()),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMinBinaryStatsBuilder<T> {
    fn stat(&self) -> AggregateFnRef {
        Stat::Min
            .aggregate_fn()
            .vortex_expect("min has an aggregate fn")
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        if let Some((lower_bound, truncated)) =
            lower_bound(T::from_scalar(value)?, self.max_value_length, nullability)
        {
            self.values.append_scalar(&lower_bound)?;
            self.is_truncated.append_value(truncated);
        } else {
            self.append_null()
        }
        Ok(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self.values.as_mut());
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![Stat::Min.name().into(), MIN_IS_TRUNCATED.into()],
            arrays: vec![
                ArrayBuilder::finish(self.values.as_mut()),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}

/// Computes the post-order sequence of `(FieldPath, DType)` entries that file-level statistics
/// are stored against.
///
/// Each leaf field (including opaque `List`/`FixedSizeList` columns, which are not recursed into)
/// gets one entry. Each **nullable** struct additionally gets a trailing entry, keyed by its own
/// path and dtype, inserted immediately after its children's entries — this carries the struct's
/// own null count. Dtypes that don't support file stats (see [`supports_file_stats`]), such as
/// [`DType::Variant`], are skipped entirely: no entry is emitted for them or anything beneath
/// them.
///
/// This function is the single source of truth for the number and order of stats entries, used
/// both when accumulating stats at write time and when reconstructing them from the footer at
/// read time, so no paths or dtypes need to be persisted in the flatbuffer itself.
pub fn postorder_stats_layout(dtype: &DType) -> Vec<(FieldPath, DType)> {
    let mut out = Vec::new();
    postorder_stats_layout_into(dtype, FieldPath::root(), &mut out);
    out
}

fn postorder_stats_layout_into(dtype: &DType, path: FieldPath, out: &mut Vec<(FieldPath, DType)>) {
    match dtype.as_struct_fields_opt() {
        Some(struct_fields) => {
            for (name, field_dtype) in struct_fields.names().iter().zip(struct_fields.fields()) {
                postorder_stats_layout_into(&field_dtype, path.clone().push(name.clone()), out);
            }
            if dtype.nullability() == Nullability::Nullable {
                out.push((path, dtype.clone()));
            }
        }
        None if !supports_file_stats(dtype) => {}
        None => out.push((path, dtype.clone())),
    }
}

/// A node in the tree of accumulators mirroring [`postorder_stats_layout`]'s walk of a `DType`.
enum StatsNode {
    /// An opaque leaf: a non-struct dtype, including `List`/`FixedSizeList` (not recursed into).
    Leaf(StatsAccumulator),
    /// A dtype that does not support file stats (e.g. [`DType::Variant`]); contributes no entries.
    Skipped,
    Struct {
        /// One child per struct field, in declaration order.
        children: Vec<(FieldName, StatsNode)>,
        /// Accumulates the struct's own null count. `Some` iff the struct itself is nullable.
        null_count: Option<StatsAccumulator>,
    },
}

impl StatsNode {
    fn build(
        dtype: &DType,
        stats: &[AggregateFnRef],
        max_variable_length_statistics_size: usize,
    ) -> Self {
        match dtype.as_struct_fields_opt() {
            Some(struct_fields) => {
                let children = struct_fields
                    .names()
                    .iter()
                    .zip(struct_fields.fields())
                    .map(|(name, field_dtype)| {
                        (
                            name.clone(),
                            Self::build(&field_dtype, stats, max_variable_length_statistics_size),
                        )
                    })
                    .collect();
                let null_count = (dtype.nullability() == Nullability::Nullable).then(|| {
                    StatsAccumulator::new(dtype, stats, max_variable_length_statistics_size)
                });
                Self::Struct {
                    children,
                    null_count,
                }
            }
            None if !supports_file_stats(dtype) => Self::Skipped,
            None => Self::Leaf(StatsAccumulator::new(
                dtype,
                stats,
                max_variable_length_statistics_size,
            )),
        }
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) => acc.push_chunk(array, ctx),
            Self::Struct { .. } => {
                let struct_array = array.clone().execute::<StructArray>(ctx)?;
                self.push_struct_chunk(array, &struct_array, ctx)
            }
        }
    }

    /// Pushes a chunk into a `Struct` node given an already-executed `StructArray` for `array`.
    ///
    /// Lets callers that already had to execute the chunk to a `StructArray` for another purpose
    /// (e.g. [`FileStatsAccumulator::process`], which also feeds the legacy per-top-level-field
    /// accumulators from the same execution) avoid doing so a second time.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not `Self::Struct`.
    fn push_struct_chunk(
        &mut self,
        array: &ArrayRef,
        struct_array: &StructArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Self::Struct {
            children,
            null_count,
        } = self
        else {
            vortex_panic!("push_struct_chunk is only called on Struct nodes");
        };
        // The struct's own `ArrayRef` already carries the validity needed to compute its null
        // count, so we push it directly rather than building a synthetic array.
        if let Some(null_count) = null_count {
            null_count.push_chunk(array, ctx)?;
        }
        for ((_, child), field) in children
            .iter_mut()
            .zip_eq(struct_array.iter_unmasked_fields())
        {
            child.push_chunk(field, ctx)?;
        }
        Ok(())
    }

    /// Appends this node's `StatsSet`s, in the same post-order as [`postorder_stats_layout`].
    fn collect_stats_sets(
        &mut self,
        ctx: &mut ExecutionCtx,
        out: &mut Vec<StatsSet>,
    ) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) => {
                out.push(acc.as_stats_set(ctx)?);
                Ok(())
            }
            Self::Struct {
                children,
                null_count,
            } => {
                for (_, child) in children.iter_mut() {
                    child.collect_stats_sets(ctx, out)?;
                }
                if let Some(null_count) = null_count {
                    out.push(null_count.as_stats_set(ctx)?);
                }
                Ok(())
            }
        }
    }
}

/// An array stream processor that computes aggregate statistics for every field, recursing into
/// nested (possibly nullable) structs. See [`postorder_stats_layout`] for the entry ordering.
///
/// Alongside `root`'s post-order nested layout, this also maintains a `legacy` accumulation
/// covering only the top-level struct fields (or the whole array, for a non-struct root),
/// matching the layout file stats had before nested struct support: one entry per top-level
/// field, with no recursion into nested structs. `legacy` is empty (and never accumulated into)
/// when `write_legacy_stats` is `false`.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    root: Arc<Mutex<StatsNode>>,
    legacy: Arc<Mutex<Vec<StatsAccumulator>>>,
    write_legacy_stats: bool,
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl FileStatsAccumulator {
    fn new(
        dtype: &DType,
        stats: Arc<[AggregateFnRef]>,
        max_variable_length_statistics_size: usize,
        session: &VortexSession,
        write_legacy_stats: bool,
    ) -> Self {
        let root = Arc::new(Mutex::new(StatsNode::build(
            dtype,
            &stats,
            max_variable_length_statistics_size,
        )));

        let legacy = Arc::new(Mutex::new(if write_legacy_stats {
            match dtype.as_struct_fields_opt() {
                Some(struct_fields) => struct_fields
                    .fields()
                    .map(|field_dtype| {
                        StatsAccumulator::new(
                            &field_dtype,
                            &stats,
                            max_variable_length_statistics_size,
                        )
                    })
                    .collect(),
                None => vec![StatsAccumulator::new(
                    dtype,
                    &stats,
                    max_variable_length_statistics_size,
                )],
            }
        } else {
            Vec::new()
        }));

        Self {
            root,
            legacy,
            write_legacy_stats,
            ctx: Arc::new(Mutex::new(session.create_execution_ctx())),
        }
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        let mut ctx = self.ctx.lock();
        if chunk.dtype().is_struct() {
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            if self.write_legacy_stats {
                for (acc, field) in self
                    .legacy
                    .lock()
                    .iter_mut()
                    .zip_eq(struct_chunk.iter_unmasked_fields())
                {
                    acc.push_chunk(field, &mut ctx)?;
                }
            }
            self.root
                .lock()
                .push_struct_chunk(&chunk, &struct_chunk, &mut ctx)?;
        } else {
            if self.write_legacy_stats {
                self.legacy.lock()[0].push_chunk(&chunk, &mut ctx)?;
            }
            self.root.lock().push_chunk(&chunk, &mut ctx)?;
        }
        Ok((sequence_id, chunk))
    }

    pub fn stats_sets(&self) -> Vec<StatsSet> {
        let mut ctx = self.ctx.lock();
        let mut out = Vec::new();
        self.root
            .lock()
            .collect_stats_sets(&mut ctx, &mut out)
            .vortex_expect("collect_stats_sets should not fail");
        out
    }

    /// Returns the legacy top-level-fields-only stats sets (one per top-level struct field, or a
    /// single entry for a non-struct root dtype). Empty if `write_legacy_stats` was `false`.
    pub fn legacy_stats_sets(&self) -> Vec<StatsSet> {
        let mut ctx = self.ctx.lock();
        self.legacy
            .lock()
            .iter_mut()
            .map(|acc| {
                acc.as_stats_set(&mut ctx)
                    .vortex_expect("as_stats_set should not fail")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::ScalarValue;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use super::*;

    fn agg(stat: Stat) -> AggregateFnRef {
        stat.aggregate_fn()
            .vortex_expect("test only uses stats with an aggregate fn")
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncates_accumulated_stats(#[case] dtype: DType) {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype.clone(),
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value("Value to be truncated");
        builder.append_value("untruncated");
        let mut builder2 = VarBinViewBuilder::with_capacity_in(
            dtype,
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder2.append_value("Another");
        builder2.append_value("wait a minute");
        let mut acc = StatsAccumulator::new(
            builder.dtype(),
            &[agg(Stat::Max), agg(Stat::Min), agg(Stat::Sum)],
            12,
        );
        acc.push_chunk(&builder.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");
        acc.push_chunk(&builder2.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");
        let stats_table = acc
            .truncated_array(&mut ctx)
            .unwrap()
            .expect("Must have stats table");
        assert_eq!(
            stats_table.names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
            ]
        );
        let field1_bool = stats_table
            .unmasked_field(1)
            .clone()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            field1_bool.to_bit_buffer(),
            BitBuffer::from(vec![false, true])
        );
        let field3_bool = stats_table
            .unmasked_field(3)
            .clone()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            field3_bool.to_bit_buffer(),
            BitBuffer::from(vec![true, false])
        );
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncated_accumulated_stats_are_inexact(#[case] dtype: DType) {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype,
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value("Value to be truncated");
        builder.append_value("Another truncated value");
        let mut acc = StatsAccumulator::new(builder.dtype(), &[agg(Stat::Max), agg(Stat::Min)], 12);
        acc.push_chunk(&builder.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");

        let stats = acc
            .as_stats_set(&mut ctx)
            .vortex_expect("as_stats_set should succeed for test data");

        assert!(matches!(stats.get(Stat::Min), Precision::Inexact(_)));
        assert!(matches!(stats.get(Stat::Max), Precision::Inexact(_)));
    }

    #[test]
    fn fixed_width_stats_omit_is_truncated_columns() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![0, 1, 2].into_array();
        let mut acc = StatsAccumulator::new(
            array.dtype(),
            &[agg(Stat::Max), agg(Stat::Min), agg(Stat::Sum)],
            12,
        );
        acc.push_chunk(&array, &mut ctx)
            .vortex_expect("push_chunk should succeed for test array");

        // Fixed-width Min/Max/Sum are AggRef-backed now, so there's no intermediate
        // truncated-stats table at all — they resolve directly through `as_stats_set`.
        assert!(acc.truncated_array(&mut ctx)?.is_none());

        let stats = acc.as_stats_set(&mut ctx)?;
        assert_eq!(
            stats.get(Stat::Max).as_exact(),
            Some(ScalarValue::from(2i32))
        );
        assert_eq!(
            stats.get(Stat::Min).as_exact(),
            Some(ScalarValue::from(0i32))
        );
        assert_eq!(
            stats.get(Stat::Sum).as_exact(),
            Some(ScalarValue::from(3i64))
        );
        Ok(())
    }

    #[test]
    fn aggregates_persist_across_multiple_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = i32_dtype();
        let mut acc = StatsAccumulator::new(
            &dtype,
            &[
                agg(Stat::Max),
                agg(Stat::Min),
                agg(Stat::Sum),
                agg(Stat::NullCount),
            ],
            12,
        );

        acc.push_chunk(&buffer![0, 5, 2].into_array(), &mut ctx)?;
        acc.push_chunk(&buffer![7, 1, 3].into_array(), &mut ctx)?;
        acc.push_chunk(&buffer![-4, 9].into_array(), &mut ctx)?;

        let stats = acc.as_stats_set(&mut ctx)?;
        assert_eq!(
            stats.get(Stat::Max).as_exact(),
            Some(ScalarValue::from(9i32))
        );
        assert_eq!(
            stats.get(Stat::Min).as_exact(),
            Some(ScalarValue::from(-4i32))
        );
        assert_eq!(
            stats.get(Stat::Sum).as_exact(),
            Some(ScalarValue::from(23i64))
        );
        assert_eq!(
            stats.get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(0u64))
        );
        Ok(())
    }

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    #[test]
    fn postorder_layout_flat_struct() {
        let dtype = DType::struct_(
            [
                ("a", i32_dtype()),
                ("b", DType::Bool(Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );
        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a"), i32_dtype()),
                (
                    FieldPath::from_name("b"),
                    DType::Bool(Nullability::Nullable)
                ),
            ]
        );
    }

    #[test]
    fn postorder_layout_nested_nullable_struct_trails_children() {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let dtype = DType::struct_([("a", inner.clone())], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a").push("b"), i32_dtype()),
                (FieldPath::from_name("a"), inner),
            ]
        );
    }

    #[test]
    fn postorder_layout_non_nullable_nested_struct_has_no_own_entry() {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::NonNullable);
        let dtype = DType::struct_([("a", inner)], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![(FieldPath::from_name("a").push("b"), i32_dtype())]
        );
    }

    #[test]
    fn postorder_layout_nullable_root_struct_gets_trailing_root_entry() {
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::Nullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a"), i32_dtype()),
                (FieldPath::root(), dtype),
            ]
        );
    }

    #[test]
    fn postorder_layout_list_field_is_opaque_leaf() {
        let list_dtype = DType::list(i32_dtype(), Nullability::NonNullable);
        let dtype = DType::struct_([("a", list_dtype.clone())], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(layout, vec![(FieldPath::from_name("a"), list_dtype)]);
    }

    #[test]
    fn postorder_layout_variant_field_is_skipped() {
        let dtype = DType::struct_(
            [
                ("a", i32_dtype()),
                ("v", DType::Variant(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(layout, vec![(FieldPath::from_name("a"), i32_dtype())]);
    }

    #[test]
    fn postorder_layout_three_level_nesting_with_mixed_nullability() {
        // root (nullable) -> a (non-nullable) -> b (nullable) -> c (leaf). Exercises composition
        // across more than one level, including a non-nullable struct sandwiched between two
        // nullable ones, which should get no entry of its own.
        let b_dtype = DType::struct_([("c", i32_dtype())], Nullability::Nullable);
        let a_dtype = DType::struct_([("b", b_dtype.clone())], Nullability::NonNullable);
        let root_dtype = DType::struct_([("a", a_dtype)], Nullability::Nullable);

        let layout = postorder_stats_layout(&root_dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a").push("b").push("c"), i32_dtype()),
                (FieldPath::from_name("a").push("b"), b_dtype),
                (FieldPath::root(), root_dtype),
            ]
        );
    }

    #[test]
    fn three_level_nested_struct_accumulates_stats_at_each_level() -> VortexResult<()> {
        // Same shape as `postorder_layout_three_level_nesting_with_mixed_nullability`, but
        // exercises actual accumulation: each nullable level along the path should independently
        // contribute its own null-count entry, in post-order.
        let mut ctx = array_session().create_execution_ctx();

        let leaf_c = buffer![10i32, 20, 30].into_array();
        let b_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let struct_b =
            StructArray::new(FieldNames::from(["c"]), [leaf_c], 3, b_validity).into_array();
        let struct_a = StructArray::new(
            FieldNames::from(["b"]),
            [struct_b],
            3,
            Validity::NonNullable,
        )
        .into_array();
        let root_validity = Validity::Array(BoolArray::from_iter([true, true, false]).into_array());
        let root =
            StructArray::new(FieldNames::from(["a"]), [struct_a], 3, root_validity).into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)];
        let mut node = StatsNode::build(root.dtype(), &requested, 1024);
        node.push_chunk(&root, &mut ctx)?;

        let mut stats_sets = Vec::new();
        node.collect_stats_sets(&mut ctx, &mut stats_sets)?;

        // `a.b.c`'s own stats come first (post-order), then `a.b`'s null-count entry, then the
        // root's own null-count entry.
        assert_eq!(stats_sets.len(), 3);
        assert_eq!(
            stats_sets[0].get(Stat::Min).as_exact(),
            Some(ScalarValue::from(10i32))
        );
        assert_eq!(
            stats_sets[0].get(Stat::Max).as_exact(),
            Some(ScalarValue::from(30i32))
        );
        assert_eq!(
            stats_sets[0].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(0u64))
        );
        assert_eq!(
            stats_sets[1].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        assert_eq!(
            stats_sets[2].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        Ok(())
    }

    #[test]
    fn nested_nullable_struct_accumulates_its_own_null_count() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)];
        let mut node = StatsNode::build(outer.dtype(), &requested, 1024);
        node.push_chunk(&outer, &mut ctx)?;

        let mut stats_sets = Vec::new();
        node.collect_stats_sets(&mut ctx, &mut stats_sets)?;

        // `a.b`'s stats come first (post-order), then `a`'s own null-count entry.
        assert_eq!(stats_sets.len(), 2);
        assert_eq!(
            stats_sets[1].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::Primitive(PValue::U64(1)))
        );
        Ok(())
    }

    #[test]
    fn legacy_stats_sets_covers_only_top_level_fields() -> VortexResult<()> {
        // The legacy accumulation must have exactly one entry per top-level field, built against
        // that field's own dtype without recursing into it, even though it's a nested struct.
        let session = array_session();
        let inner_dtype = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let dtype = DType::struct_([("a", inner_dtype)], Nullability::NonNullable);

        let acc = FileStatsAccumulator::new(
            &dtype,
            Arc::from([agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)]),
            1024,
            &session,
            true,
        );

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        let (mut ptr, _eof) = SequenceId::root().split();
        acc.process(Ok((ptr.advance(), outer)))?;

        let legacy = acc.legacy_stats_sets();
        assert_eq!(legacy.len(), 1);
        assert_eq!(
            legacy[0].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        assert!(legacy[0].get(Stat::Min).as_exact().is_none());
        assert!(legacy[0].get(Stat::Max).as_exact().is_none());
        Ok(())
    }

    #[rstest]
    #[case(DType::struct_([("a", i32_dtype())], Nullability::NonNullable), StructArray::new(FieldNames::from(["a"]), [buffer![1i32, 2, 3].into_array()], 3, Validity::NonNullable).into_array())]
    #[case(i32_dtype(), buffer![1i32, 2, 3].into_array())]
    fn write_legacy_stats_false_skips_legacy_accumulation(
        #[case] dtype: DType,
        #[case] chunk: ArrayRef,
    ) -> VortexResult<()> {
        // With `write_legacy_stats: false`, `legacy` stays empty for both struct and non-struct
        // roots. This must not panic — `process` indexes into `legacy[0]` for a non-struct root
        // and `zip_eq`s it against the struct's fields otherwise, both of which would panic if
        // `legacy` were left empty without also gating those code paths.
        let session = array_session();
        let acc = FileStatsAccumulator::new(
            &dtype,
            Arc::from([agg(Stat::Min), agg(Stat::Max)]),
            1024,
            &session,
            false,
        );

        let (mut ptr, _eof) = SequenceId::root().split();
        acc.process(Ok((ptr.advance(), chunk)))?;

        assert!(acc.legacy_stats_sets().is_empty());
        Ok(())
    }
}
