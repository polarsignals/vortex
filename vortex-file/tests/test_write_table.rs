// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::FixedSizeBinaryArray;
use arrow_array::RecordBatch;
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::StreamExt;
use futures::pin_mut;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_dtype::FieldNames;
use vortex_dtype::field_path;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&mut session);

    session
});

#[tokio::test]
async fn test_file_roundtrip() {
    // Create a simple roundtrip
    let nums = PrimitiveArray::from_iter((0..1024).cycle().take(16_384)).into_array();

    let a_array = StructArray::new(
        FieldNames::from(["raw", "compressed"]),
        vec![nums.clone(), nums.clone()],
        16_384,
        Validity::NonNullable,
    )
    .into_array();

    let b_array = PrimitiveArray::from_iter((1024..2048).cycle().take(16_384)).into_array();

    let data = StructArray::new(
        FieldNames::from(["a", "b"]),
        vec![a_array, b_array],
        16_384,
        Validity::NonNullable,
    )
    .into_array();

    // Create a writer which by default uses the BtrBlocks compressor for a.compressed, but leaves
    // the b and the a.raw columns uncompressed.
    let default_strategy = Arc::new(CompressingStrategy::new_btrblocks(
        FlatLayoutStrategy::default(),
        false,
    ));

    let writer = Arc::new(
        TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), default_strategy)
            .with_field_writer(field_path!(a.raw), Arc::new(FlatLayoutStrategy::default()))
            .with_field_writer(field_path!(b), Arc::new(FlatLayoutStrategy::default())),
    );

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(writer)
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    // Read the data back
    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");

    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        let next = next.expect("next");
        let next = next.to_struct();
        let a = next.unmasked_field_by_name("a").unwrap().to_struct();
        let b = next.unmasked_field_by_name("b").unwrap();

        let raw = a.unmasked_field_by_name("raw").unwrap();
        let compressed = a.unmasked_field_by_name("compressed").unwrap();

        assert!(raw.is_canonical());
        assert!(!compressed.is_canonical());

        assert!(b.is_canonical());
        assert!(raw.nbytes() > compressed.nbytes());
    }
}

/// Regression test: writing a Dict<ListView> where the list has
/// Validity::Array(BoolArray) and the dict codes are nullable used to fail
/// with "Array vortex.fill_null does not support serialization".
#[tokio::test]
async fn test_dict_listview_validity_roundtrip() {
    let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]).into_array();
    let offsets = PrimitiveArray::from_iter(vec![0u32, 2, 4]).into_array();
    let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 1]).into_array();
    let list_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
    let listview = ListViewArray::new(elements, offsets, sizes, list_validity).into_array();

    let codes = PrimitiveArray::new(
        vortex_buffer::buffer![0u32, 0, 1, 0, 2],
        Validity::from_iter(vec![true, false, true, true, true]),
    )
    .into_array();

    let dict = DictArray::new(codes, listview).into_array();

    let data = StructArray::from_fields(&[("col", dict)])
        .expect("from_fields")
        .into_array();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write should not fail with fill_null serialization error");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let chunk = stream
        .next()
        .await
        .unwrap()
        .expect("read back should succeed");
    vortex_array::assert_arrays_eq!(data, chunk);
    assert!(stream.next().await.is_none(), "expected a single chunk");
}

/// Roundtrip test for UUID extension type through the Vortex file format.
///
/// Creates an Arrow RecordBatch with a UUID-annotated FixedSizeBinary(16) column,
/// converts to Vortex, writes to a Vortex file, reads back, and asserts equality.
#[tokio::test]
async fn test_uuid_roundtrip() {
    // Build some UUID bytes (mix of values and nulls).
    let uuid1 = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
    let uuid2 = uuid::Uuid::parse_str("6ba7b810-9dad-11d1-80b4-00c04fd430c8").unwrap();
    let uuid3 = uuid::Uuid::parse_str("f47ac10b-58cc-4372-a567-0e02b2c3d479").unwrap();

    let uuid_bytes: Vec<Option<&[u8]>> = vec![
        Some(uuid1.as_bytes()),
        None,
        Some(uuid2.as_bytes()),
        Some(uuid3.as_bytes()),
        None,
    ];

    let fsb_array = FixedSizeBinaryArray::from(uuid_bytes);

    let field = Field::new("uuids", arrow_schema::DataType::FixedSizeBinary(16), true)
        .with_extension_type(arrow_schema::extension::Uuid);
    let schema = Schema::new(vec![field]);
    let record_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(fsb_array)]).unwrap();

    let data = ArrayRef::from_arrow(&record_batch, false).unwrap();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let chunk = stream
        .next()
        .await
        .unwrap()
        .expect("read back should succeed");
    vortex_array::assert_arrays_eq!(data, chunk);
    assert!(stream.next().await.is_none(), "expected a single chunk");
}
