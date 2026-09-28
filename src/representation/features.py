def calc_drive(frame):
    return {
        "velocity_mean": frame["Velocity"].mean(),
        "velocity_sd": frame["Velocity"].std(ddof=1),

        "lane_offset_mean": frame["Lane Offset"].mean(),
        "lane_offset_sd": frame["Lane Offset"].std(ddof=1),
        "lane_offset_abs_mean": frame["Lane Offset"].abs().mean(),

        "acc_mean": frame["Acceleration"].mean(),
        "acc_sd": frame["Acceleration"].std(ddof=1),
    }