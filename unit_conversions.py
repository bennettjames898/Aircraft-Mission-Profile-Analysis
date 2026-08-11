"""
Unit conversons used throughout the project.

"""
# Convert FEET to METER
def ft_to_m(ft: float) -> float:
    return ft * 0.3048

# Convert METER to FEET
def m_to_ft(m: float) -> float:
    return m / 0.3048

# Convert KNOT to METER PER SEC
def kt_to_ms(kt: float) -> float:
    return kt * 0.514444

# Convert METER PER SEC to KNOT
def ms_to_kt(ms: float) -> float:
    return ms / 0.514444