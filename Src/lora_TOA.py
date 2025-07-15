import math

def lora_time_on_air(sf, bw, pl, cr=1, preamble_len=8, has_crc=True, has_header=True):
    # Symbol duration
    t_sym = (2 ** sf) / bw

    # Low Data Rate Optimization
    de = 1 if sf >= 11 else 0

    # Payload symbol calculation
    h = 0 if has_header else 1
    crc = 1 if has_crc else 0

    payload_symb_nb = 8 + max(
        math.ceil(
            (8 * pl - 4 * sf + 28 + 16 * crc - 20 * h)
            / (4 * (sf - 2 * de))
        ) * (cr + 4),
        0,
    )
    # Total time on air
    t_preamble = (preamble_len + 4.25) * t_sym
    t_payload = payload_symb_nb * t_sym
    t_on_air = t_preamble + t_payload

    return t_on_air * 1000  # in milliseconds

# ==== User Input ====
try:
    print("LoRa Time-on-Air Calculator (EU868 Region)")
    sf = int(input("Enter Spreading Factor (SF7 to SF12): "))
    bw_khz = int(input("Enter Bandwidth in kHz (typically 125 for EU): "))
    pl = int(input("Enter Payload size (in bytes): "))

    if sf not in range(7, 13):
        raise ValueError("SF must be between 7 and 12")
    if bw_khz not in [125, 250, 500]:
        raise ValueError("BW should typically be 125, 250, or 500 kHz")

    bw = bw_khz * 1000  # convert to Hz

    toa = lora_time_on_air(sf, bw, pl)
    print(f"\n🕒 Estimated Time-on-Air: {toa:.2f} ms")

except Exception as e:
    print(f"Error: {e}")
