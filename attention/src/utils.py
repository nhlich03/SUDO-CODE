def epoch_time(start_time, end_time):
    import time
    elapsed_time = end_time - start_time
    mins, secs = divmod(int(elapsed_time), 60)
    return mins, secs
