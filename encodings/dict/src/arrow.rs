// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::types::ArrowDictionaryKeyType;
use arrow_array::{make_array, AnyDictionaryArray, ArrayRef as ArrowArrayRef, DictionaryArray};
use arrow_data::ArrayDataBuilder;
use arrow_schema::DataType;
use vortex_array::arrow::compute::{ToArrowKernel, ToArrowKernelAdapter};
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::{register_kernel, ArrayRef};
use vortex_error::{VortexResult, VortexUnwrap};

use crate::{DictArray, DictVTable};

impl<K: ArrowDictionaryKeyType> FromArrowArray<&DictionaryArray<K>> for DictArray {
    fn from_arrow(array: &DictionaryArray<K>, nullable: bool) -> Self {
        let keys = AnyDictionaryArray::keys(array);
        let keys = ArrayRef::from_arrow(keys, keys.is_nullable());
        let values = ArrayRef::from_arrow(array.values().as_ref(), nullable);
        DictArray::try_new(keys, values).vortex_unwrap()
    }
}

register_kernel!(ToArrowKernelAdapter(DictVTable).lift());

impl ToArrowKernel for DictVTable {
    fn to_arrow(
        &self,
        array: &DictArray,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        let (arrow_keys, arrow_values) = match arrow_type {
            None => (
                array.codes().clone().into_arrow_preferred()?,
                array.values().clone().into_arrow_preferred()?,
            ),
            Some(DataType::Dictionary(codes_type, values_type)) => (
                array.codes().clone().into_arrow(codes_type)?,
                array.values().clone().into_arrow(values_type)?,
            ),
            _ => {
                // Unsupported type.
                return Ok(None);
            }
        };
        let keys_data = arrow_keys.to_data();
        Ok(Some(make_array(
            ArrayDataBuilder::new(DataType::Dictionary(
                Box::new(arrow_keys.data_type().clone()),
                Box::new(arrow_values.data_type().clone()),
            ))
            .len(keys_data.len())
            .add_buffers(keys_data.buffers().iter().cloned())
            .nulls(keys_data.nulls().cloned())
            .add_child_data(arrow_values.to_data())
            .build()?,
        )))
    }
}
