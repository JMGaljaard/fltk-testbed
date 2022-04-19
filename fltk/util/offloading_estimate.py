
def calc_optimal_offloading_point(profiler_data, time_till_deadline, iterations_left):
    ff, cf, cb, fb = profiler_data
    full_network = ff + cf + cb + fb
    frozen_network = ff + cf + cb
    split_point = 0
    for z in range(iterations_left, -1, -1):
        x = z
        y = iterations_left - x
        # print(z)
        new_est_split = (x * full_network) + (y * frozen_network)
        split_point = x
        if new_est_split < time_till_deadline:
            break


def estimate():
    """
    freeze_network = ff + cf + cb + fb
    frozen_network = ff + cf + cb

    td = time until deadline
    cl = cycles left

    a = 1
    b = cl - a


    """
    np = {
        'a': 2,
        'b': 1,
        'c': 3,
        'd': 4,
    }

    sp = {
        'time_left': 400,
        'iter_left': 44
    }

    f_n = np['a'] + np['b'] + np['c'] + np['d']
    o_n = np['a'] + np['b'] + np['c']
    est_full_comp_time = f_n * sp['iter_left']
    new_est = o_n * sp['iter_left']
    x = 20
    y = sp['iter_left'] - x
    new_est_split = (x * f_n) + (y * o_n)

    print(f'estimate: {est_full_comp_time} < {sp["time_left"]} ? {est_full_comp_time <sp["time_left"]}')
    print(f'new estimate: {new_est} < {sp["time_left"]} ? {new_est <sp["time_left"]}')
    for z in range(sp['iter_left']):
        x = z
        y = sp['iter_left'] - x
        # print(z)
        new_est_split = (x * f_n) + (y * o_n)
        print(f'new new_est_split(x={x}, y={y}): {new_est_split} < {sp["time_left"]} ? {new_est_split <sp["time_left"]}')

if __name__ == '__main__':
    print('Starting estimate')
    estimate()