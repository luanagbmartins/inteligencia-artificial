import subprocess
import multiprocessing

def run(cmd):
    return subprocess.call(cmd)

if __name__ == '__main__':

    while True:
        cmds_list = [ ['python', 'trees.py'] for i in range(16) ] 
        p = multiprocessing.Pool(16)
        p.map(run, cmds_list)
        p.close()
        p.join()

        cmds_list = [ ['python', 'bayes.py'] for i in range(32) ] 
        p = multiprocessing.Pool(16)
        p.map(run, cmds_list)
        p.close()
        p.join()

        cmds_list = [ ['python', 'ann.py'] for i in range(4) ] 
        p = multiprocessing.Pool(4)
        p.map(run, cmds_list)
        p.close()
        p.join()

        cmds_list = [ ['python', 'ann_trees.py'] for i in range(4) ] 
        p = multiprocessing.Pool(4)
        p.map(run, cmds_list)
        p.close()
        p.join()