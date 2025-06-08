def compose_print(x):
    return f"\033[{x}m"


PRINT_RESUME = compose_print(0)
PRINT_ACTION = compose_print(31) + compose_print(42)
PRINT_STEP = compose_print(1) + compose_print(46)
PRINT_CURRENT = compose_print(4) + compose_print(45)


def print_action(*args):
    args_str = "".join(args)
    print_str = f"{PRINT_ACTION} {args_str} {PRINT_RESUME}"
    print(PRINT_ACTION, *args, PRINT_RESUME)
    return print_str


def print_step(*args):
    args_str = "".join(args)
    print_str = f"{PRINT_STEP} {args_str} {PRINT_RESUME}"
    print(PRINT_STEP, *args, PRINT_RESUME)
    return print_str


def print_current(*args):
    args_str="".join(args)
    print_str=f"{PRINT_CURRENT} {args_str} {PRINT_RESUME}"
    print(PRINT_CURRENT, *args, PRINT_RESUME)
    return print_str
