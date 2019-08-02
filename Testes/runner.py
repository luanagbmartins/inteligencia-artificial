import subprocess
import multiprocessing

def run(cmd):
    return subprocess.call(cmd)

if __name__ == '__main__':
    CORES = 16
    cmds_list = [ ['python', 'bayes.py'] for i in range(16) ] 
    p = multiprocessing.Pool(16)
    p.map(run, cmds_list)
    p.close()
    p.join()
