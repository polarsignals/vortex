// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeBinaryArray;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::vtable::ValidityHelper;

/// Convert a Vortex extension array (e.g. UUID) to an Arrow `FixedSizeBinaryArray`.
///
/// The array must be an extension type whose storage is `FixedSizeList(Primitive(U8), size)`.
pub(super) fn to_arrow_fixed_size_binary(
    array: ArrayRef,
    size: i32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let Some(ext) = array.dtype().as_extension_opt() else {
        vortex_bail!(
            "FixedSizeBinary conversion requires an extension dtype, got {}",
            array.dtype()
        );
    };

    match ext.storage_dtype() {
        DType::FixedSizeList(elem, list_size, _)
            if *list_size == size as u32
                && matches!(elem.as_ref(), DType::Primitive(PType::U8, _)) => {}
        other => {
            vortex_bail!(
                "FixedSizeBinary({size}) conversion requires FixedSizeList(U8, {size}) storage, got {other}"
            );
        }
    }

    let ext_array = array.execute::<ExtensionArray>(ctx)?;
    let fsl = ext_array
        .storage()
        .clone()
        .execute::<FixedSizeListArray>(ctx)?;
    let elements = fsl.elements().clone().execute::<PrimitiveArray>(ctx)?;
    let values = elements.into_buffer::<u8>().into_arrow_buffer();
    let null_buffer = to_arrow_null_buffer(fsl.validity().clone(), fsl.len(), ctx)?;

    Ok(Arc::new(FixedSizeBinaryArray::new(
        size,
        values,
        null_buffer,
    )))
}
