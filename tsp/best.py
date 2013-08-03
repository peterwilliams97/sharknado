import imp, glob, os, math
from collections import defaultdict



def trip(points, order):
    #print type(points)
    #print len(points)
    dist = 0
    p0x, p0y = points[order[-1]]
    for i in order:
        p1x, p1y = points[i]
        dist += math.sqrt((p1x-p0x) ** 2 + (p1y-p0y) ** 2)  
        p0x, p0y = p1x, p1y    
    return dist
    

 

histories = list(glob.glob('history*.py'))

for filepath in histories:
    print filepath

all_solutions = defaultdict(dict)
all_points = defaultdict(dict)
   
for filepath in histories:
    mod_name = os.path.splitext(os.path.split(filepath)[-1])[0]    
    modul = imp.load_source(mod_name, filepath)
    #print mod_name, modul.saved_solutions.keys(), modul.saved_points.keys()
    for name, soln in modul.saved_solutions.items():
        #print name, mod_name, 
        all_solutions[name][mod_name] = soln
        all_points[name][mod_name] = modul.saved_points[name]    


HISTORY = 'best_history.py'   
#saved_paths = set() 
saved_points = {}    
saved_solutions = {}         
saved_scores = {}
         
for name in sorted(all_solutions):
    best = [(mod_name, soln[0]) for mod_name, soln in all_solutions[name].items()]
    best = sorted(all_solutions[name].keys(), key=lambda k: all_solutions[name][k][0])
    #best.sort(key=lambda x: x[1])
    print '-' * 80
    print name
    for hist in best:
        score = all_solutions[name][hist][0]
        order = all_solutions[name][hist][1]
        points = all_points[name][hist]
        actual_score = trip(points, order)
        assert abs(score - actual_score) < 1e-5, '%s %s %f %f %f' % (hist, name, 
                score, actual_score, score - actual_score)
        print hist, score, actual_score
    
    saved_solutions[name] = all_solutions[name][best[0]]
    saved_points[name] = all_points[name][best[0]]
    saved_scores[name] = all_solutions[name][best[0]][0]
   
   
history = 'best_history.py'
print 'Writing history:', history
with open(history, 'wt') as f:
    f.write('from numpy import array\n\n')
    f.write('VERSION=%d\n' % 9999)
    #f.write('MAX_CLOSEST=%d\n' % MAX_CLOSEST)
    #f.write('MAX_N=%d\n' % MAX_N)
    #f.write('RANDOM_SEED=%d\n' % RANDOM_SEED)
    #f.write('DEBUG=%s\n' % DEBUG)
    #f.write('EPSILON=%s\n' % EPSILON)
    f.write('saved_paths = %s\n' % repr(sorted(saved_solutions.keys())))
    f.write('saved_scores = %s\n' % repr(saved_scores))
    f.write('saved_solutions = %s\n' % repr(saved_solutions))
    f.write('saved_points = %s\n' % repr(saved_points))   
    