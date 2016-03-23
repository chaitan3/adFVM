import argparse
p = argparse.ArgumentParser()
p.add_argument('case')
p.add_argument('time', nargs='+', type=float)
user = p.parse_args()
print user.time
