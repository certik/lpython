def main0():
    i: i32
    sum: i32
    sum = 0
    for i in range(0, 10):
        if i == 0:
            continue
        sum += i
        if i > 5:
            break
    assert sum == 21

main0()
