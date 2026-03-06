// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::DType;
use crate::ExtID;
use crate::PType;
use crate::extension::ExtDTypeVTable;
use crate::uuid::Uuid;
use crate::uuid::UuidMetadata;
use crate::uuid::metadata::u8_to_version;

/// The number of bytes in a UUID.
const UUID_BYTE_LEN: usize = 16;

impl ExtDTypeVTable for Uuid {
    type Metadata = UuidMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref("vortex.uuid")
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        match metadata.version {
            None => Ok(Vec::new()),
            Some(v) => Ok(vec![v as u8]),
        }
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        let version = match metadata.len() {
            0 => None,
            1 => Some(u8_to_version(metadata[0])?),
            other => vortex_bail!("UUID metadata must be 0 or 1 bytes, got {other}"),
        };

        Ok(UuidMetadata { version })
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        let DType::FixedSizeList(element_dtype, list_size, _nullability) = storage_dtype else {
            vortex_bail!("UUID storage dtype must be a FixedSizeList, got {storage_dtype}");
        };

        vortex_ensure_eq!(
            *list_size as usize,
            UUID_BYTE_LEN,
            "UUID storage FixedSizeList must have size {UUID_BYTE_LEN}, got {list_size}"
        );

        let DType::Primitive(ptype, elem_nullability) = element_dtype.as_ref() else {
            vortex_bail!("UUID element dtype must be Primitive(U8), got {element_dtype}");
        };

        vortex_ensure_eq!(
            *ptype,
            PType::U8,
            "UUID element dtype must be U8, got {ptype}"
        );
        vortex_ensure!(
            !elem_nullability.is_nullable(),
            "UUID element dtype must be non-nullable"
        );

        Ok(())
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "UUID_BYTE_LEN always fits both usize and u32"
)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use uuid::Version;
    use vortex_error::VortexResult;

    use super::UUID_BYTE_LEN;
    use crate::DType;
    use crate::Nullability;
    use crate::PType;
    use crate::extension::ExtDTypeVTable;
    use crate::uuid::Uuid;
    use crate::uuid::UuidMetadata;

    #[rstest]
    #[case::no_version(None)]
    #[case::v4_random(Some(Version::Random))]
    #[case::v7_sort_rand(Some(Version::SortRand))]
    #[case::nil(Some(Version::Nil))]
    #[case::max(Some(Version::Max))]
    fn roundtrip_metadata(#[case] version: Option<Version>) -> VortexResult<()> {
        let metadata = UuidMetadata { version };
        let bytes = Uuid.serialize(&metadata)?;
        let expected_len = if version.is_none() { 0 } else { 1 };
        assert_eq!(bytes.len(), expected_len);
        let deserialized = Uuid.deserialize(&bytes)?;
        assert_eq!(deserialized, metadata);
        Ok(())
    }

    #[test]
    fn metadata_display_no_version() {
        let metadata = UuidMetadata { version: None };
        assert_eq!(metadata.to_string(), "");
    }

    #[test]
    fn metadata_display_with_version() {
        let metadata = UuidMetadata {
            version: Some(Version::Random),
        };
        assert_eq!(metadata.to_string(), "v4");

        let metadata = UuidMetadata {
            version: Some(Version::SortRand),
        };
        assert_eq!(metadata.to_string(), "v7");
    }

    #[rstest]
    #[case::non_nullable(Nullability::NonNullable)]
    #[case::nullable(Nullability::Nullable)]
    fn validate_correct_storage_dtype(#[case] nullability: Nullability) -> VortexResult<()> {
        let metadata = UuidMetadata::default();
        let storage_dtype = uuid_storage_dtype(nullability);
        Uuid.validate_dtype(&metadata, &storage_dtype)
    }

    #[test]
    fn validate_rejects_wrong_list_size() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            8,
            Nullability::NonNullable,
        );
        assert!(
            Uuid.validate_dtype(&UuidMetadata::default(), &storage_dtype)
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_wrong_element_type() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        );
        assert!(
            Uuid.validate_dtype(&UuidMetadata::default(), &storage_dtype)
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_nullable_elements() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        );
        assert!(
            Uuid.validate_dtype(&UuidMetadata::default(), &storage_dtype)
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_non_fsl() {
        let storage_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        assert!(
            Uuid.validate_dtype(&UuidMetadata::default(), &storage_dtype)
                .is_err()
        );
    }

    fn uuid_storage_dtype(nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            nullability,
        )
    }
}
