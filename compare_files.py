with open('foo.txt', 'r') as f1, open('foo2.txt', 'r') as f2:
    # read the lines of each file into separate lists
    lines1 = f1.readlines()
    lines2 = f2.readlines()

    # compare the lines and output any differences
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            print(i, line1, line2)
