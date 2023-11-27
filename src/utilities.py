import sys


def huc_param_list(filename):
    with open(filename) as reader:
        for line in reader:
            data = line.split(":")
            code = data[0].strip()
            N = len(code)
            title = data[1].strip()
            name = title.lower().replace(" ", "-")
            with open(f".anek/inputs/huc{N}/{name}", "w") as writer:
                writer.write(f"huc={code}\n")
                writer.write(f"huc{N}={code}\n")
                writer.write(f"title={title}\n")
                writer.write(f"name={name}\n")


def main():
    try:
        func_name = sys.argv[1]
        args = sys.argv[2:]
    except IndexError:
        print("Provide Function name to run.", file=sys.stderr)
        exit(1)
    try:
        func = globals()[func_name]
    except KeyError:
        print("No such Function.", file=sys.stderr)
        exit(1)
    func(*args)
    return


if __name__ == '__main__':
    main()
