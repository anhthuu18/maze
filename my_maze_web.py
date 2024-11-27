from simpleai.search import CspProblem, backtrack

def constraint_func(names, values):
    return values[0] != values[1]

# Tự tô màu cho miền tây
if __name__=='__main__':
    names = ('WA', 'NT', 'Q', 'NSW', 'V','SA','T')
    domain = {'WA':  ['G'],
              'NT':  ['R', 'G', 'B'],
              'Q':   ['R', 'G', 'B'],
              'NSW': ['R', 'G', 'B'],
              'V':   ['R', 'G', 'B'],
              'SA':  ['R', 'G', 'B'],
              'T':   ['R', 'G', 'B'],
              }

    constraints = [
        (('SA', 'WA'), constraint_func),
        (('SA', 'NT'), constraint_func),
        (('SA', 'Q'), constraint_func),
        (('SA', 'NSW'), constraint_func),
        (('SA', 'V'), constraint_func),
        (('WA', 'NT'), constraint_func),
        (('NT', 'Q'), constraint_func),
        (('Q', 'NSW'), constraint_func),
        (('NSW', 'V'), constraint_func),
    ]

    problem = CspProblem(names, domain, constraints)
    output = backtrack(problem)
    print(output)
