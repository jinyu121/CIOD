from statistics import mean

aps = """
0.1 0.2 0.3
0.1 0.2 0.3 0.4 0.5 0.6
"""

hundred = 100.

aps = aps.strip().split("\n")
max_class = 0
groups = len(aps)
for ith in range(groups):
    if ":" in aps[ith]:
        aps[ith] = aps[ith].split(':')[1]
    aps[ith] = aps[ith].split()
    aps[ith] = [float(x) for x in aps[ith]]
    max_class = max(max_class, len(aps[ith]))

for x in aps:
    for y in x:
        if y > 1.0:
            hundred = 1.
        break

print("=" * 10, "Result Summary (mAP, AP, %)", "=" * 10)
for now_group, x in enumerate(aps):
    now_classes_low = max_class * now_group // groups
    now_classes_high = max_class * (now_group + 1) // groups
    d = x[:now_classes_high]
    print("{:.2f} :".format(mean(d) * hundred), end="\t")
    for y in d:
        print("{:.2f}\t".format(y * hundred), end="")
    print()

print("=" * 10, "Group Summary (mAP, AP, %)", "=" * 10)
for now_group in range(groups):
    now_classes_low = max_class * now_group // groups
    now_classes_high = max_class * (now_group + 1) // groups
    ans = []
    for x in range(now_group, groups):
        ans.append(mean(aps[x][now_classes_low:now_classes_high]) * hundred)
    print("Group {:>2} :".format(now_group), "\t->\t".join(map("{:.2f}".format, ans)))
