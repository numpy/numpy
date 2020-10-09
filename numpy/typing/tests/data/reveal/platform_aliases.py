import numpy.typing as npt

int8_codes: npt._Int8Codes
int16_codes: npt._Int16Codes
int32_codes: npt._Int32Codes
int64_codes: npt._Int64Codes
uint8_codes: npt._UInt8Codes
uint16_codes: npt._UInt16Codes
uint32_codes: npt._UInt32Codes
uint64_codes: npt._UInt64Codes
float16_codes: npt._Float16Codes
float32_codes: npt._Float32Codes
float64_codes: npt._Float64Codes
complex64_codes: npt._Complex64Codes
complex128_codes: npt._Complex128Codes

reveal_type(int8_codes)  # E: Literal
reveal_type(int16_codes)  # E: Literal
reveal_type(int32_codes)  # E: Literal
reveal_type(int64_codes)  # E: Literal
reveal_type(uint8_codes)  # E: Literal
reveal_type(uint16_codes)  # E: Literal
reveal_type(uint32_codes)  # E: Literal
reveal_type(uint64_codes)  # E: Literal
reveal_type(float16_codes)  # E: Literal
reveal_type(float32_codes)  # E: Literal
reveal_type(float64_codes)  # E: Literal
reveal_type(complex64_codes)  # E: Literal
reveal_type(complex128_codes)  # E: Literal
