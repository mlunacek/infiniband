#!/usr/bin/env python
'''
An abstraction of the IPython Parallel task interface.

Given a PBS_NODEFILE, this class launches the controller and engines via ssh
using a temporary profile.

Author: Monte Lunacek, monte.lunacek@colorado.edu

'''
import os
import subprocess
import time
import socket
import signal
import shutil

import uuid
import logging
import jinja2 as jin

from IPython import parallel

# Template constants
ipcontroller = jin.Template('''
c = get_config()
c.HubFactory.ip = '*'

''')

ipengine = jin.Template('''
c = get_config()
c.EngineFactory.timeout = 300
c.IPEngineApp.log_to_file = True
c.IPEngineApp.log_level = 30
c.EngineFactory.ip = '*'

''')

def execute_command(cmd):
    """Command that the map function calls"""
    
    import subprocess, socket
    
    pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pro.wait()
    stdout, stderr = pro.communicate()
    returnval = pro.returncode
    
    return {'host' :socket.gethostname(), 'cmd':cmd , 'returnval': returnval}
    
class LoadBalance:
    
    def __init__(self, ppn=12, loglevel=logging.ERROR):
        """Creates a profile, logger, starts engines and controllers"""
        self.directory = os.getcwd()
        self.logger = logging.getLogger('LoadBalance')
        self.logger.setLevel(loglevel)
        ch = logging.FileHandler(os.path.join(self.directory,'loadbalance.log'))
        ch.setLevel(loglevel)
        formatter = logging.Formatter('%(asctime)s:   %(message)s', datefmt='%I:%M:%S %p')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.set_ppn(ppn)

        self.profile = 'temp_' + str(uuid.uuid1())
        #self.profile = 'testing'
        self.ipengine_path()
        
        self.logger.debug('starting load balance')
        self.logger.debug(self.directory)
        self.node_list = self.pbs_nodes()
        #self.logger.debug(self.node_list)
        self.create_profile()
        self.start_controller()
        self.start_engines()
        self.create_view()
    
    def set_ppn(self,ppn):
        """Environmental variable override"""
        try:
            ppn = os.environ['PPN']
        except KeyError, e:
            pass

        self.ppn = int(ppn)

    def ipengine_path(self):    
        """Find the full path for ipengine"""

        p = subprocess.Popen(['which','ipengine'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.stdout.readlines()
        if len(res) == 0: 
            self.logger.error('Cannot find ipengine')
            exit(1)
        
        self.ipengine = res[0].strip('\n')

    def pbs_nodes(self):
        """Returns an array of nodes from the PBS_NODEFILE"""
        nodes = []
        try:
            filename = os.environ['PBS_NODEFILE']
        except KeyError, e:
            self.logger.debug('PBS_NODEFILE not found. You must have a reservation to run this file.')
            exit(1)
      
        with open(filename,'r') as file:
            for line in file:
                node_name = line.split()[0]
                if node_name not in nodes:
                    nodes.append(node_name)
        
        return nodes        
            
    def create_profile(self):
        """Calls the ipython profile create command"""
        cmd = subprocess.Popen(['ipython','profile','create','--parallel','--profile='+self.profile], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                preexec_fn=os.setsid)
        cmd.wait()

        # Append settings
        self.profile_directory = os.path.join(os.path.join(os.environ['HOME'],'.ipython'),'profile_'+ self.profile)

        tmp = ipcontroller.render({})
        with open(os.path.join(self.profile_directory,'ipcontroller_config.py'),'w') as f:
            f.write(tmp)    
    
        tmp = ipengine.render({})
        with open(os.path.join(self.profile_directory,'ipengine_config.py'),'w') as f:
            f.write(tmp)    
    
    def start_controller(self):
        """Starts the ipcontroller"""
        cmd = ['ipcontroller']
        cmd.append('--profile='+self.profile)
        cmd.append('--log-to-file')
        cmd.append('--log-level=20')
        cmd.append("--ip='*'")
        self.logger.debug(' '.join(cmd))
        self.controller = subprocess.Popen(cmd,
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        preexec_fn=os.setsid)
        time.sleep(1)
        self.wait_for_controller()
        self.logger.debug('controller has started\n')
        
    def wait_for_controller(self):   
        """Loops until the controller is ready"""  
        tic = time.time()
        while True:
            if  time.time() - tic > 30:
                break
            self.logger.debug('waiting for controller ' + str(time.time() - tic) )
            try:
                rc = parallel.Client(profile=self.profile)
                return True
            except ValueError, e:
                self.logger.debug(e)
                time.sleep(2)
            except IOError, e:
                self.logger.debug(e)
                time.sleep(2)
            except:
                time.sleep(2)
                                  
    def start_engines(self):
        """Starts and waits for the engines"""
        self.engines = []
        self.hostname = socket.gethostname()
        for node in self.node_list:
            for i in xrange(self.ppn):
                if self.hostname != node:
                    cmd = ['ssh']
                    cmd.append(node)
                    cmd.append(self.ipengine)
                else:
                    cmd = [self.ipengine]
      
                cmd.append('--profile='+self.profile)
                cmd.append('--log-to-file')
                cmd.append('--log-level=20')
                cmd.append('--work-dir=' + self.directory)
                self.logger.debug(' '.join(cmd))
                tmp = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                preexec_fn=os.setsid) 
                self.engines.append(tmp)
                time.sleep(0.1)
                
        self.wait_for_engines()
    
    def wait_for_engines(self):
        """Loops until engies have started"""
        tic = time.time()
        while True and time.time() - tic < 120:
            try:
                rc = parallel.Client(profile=self.profile) 
                if len(rc.ids) == len(self.engines):
                    self.logger.debug('Engines started ' + str(len(rc.ids)) )
                    return True
                else:
                    self.logger.debug('waiting for engines ' + str(time.time() - tic) + ' ' + str(len(rc.ids))) 
                    time.sleep(2)     
            except ValueError, e:
                self.logger.debug(e)
                time.sleep(2)
            except IOError, e:
                self.logger.debug(e)
                time.sleep(2)
    
    def remove_profile(self):
        """Removes the profile directory"""
        count = 0
        self.logger.debug('Attempting to remove profile')
        while True and count < 20:
            try:
                shutil.rmtree(self.profile_directory)
                count += 1
                self.logger.debug('profile removed')
                return True
            except OSError, e:
                self.logger.debug(e)
                time.sleep(1)
        self.logger.debug('unable to remove profile')
        return False
                
    def __del__(self):
        try:
            for engine in self.engines:
                os.killpg( engine.pid, signal.SIGINT)

            os.killpg( self.controller.pid, signal.SIGINT)
            self.remove_profile()
        except AttributeError:
            pass
    
    def create_view(self):
        """Creates the client and the load balance view"""
        self.rc = parallel.Client(profile=self.profile) 
        self.lview = self.rc.load_balanced_view() 
        self.lview.retries = 1
        self.logger.debug('Number of engines ' + str(len(self.rc)))

    def set_retries(self, num_retries):
        self.lview.retries = num_retries

    def map(self, input_func, input_list):

        number_of_jobs = len(input_list)
        self.logger.debug(number_of_jobs)

        tic = time.time()
        ar = self.lview.map(execute_command, input_list)
        #ar.wait()
        for i,r in enumerate(ar):
             self.logger.debug("task: %i finished on %s, %.3f percent finished at time %.3f "%(
                                i, r['host'], 100*((i+1)/float(number_of_jobs)), time.time()-tic ))

        self.logger.debug('done')
        return ar


    def run_commands(self, commands):
        """Maps the commands to the execute_command function, in parallel"""
        self.logger.debug('running')
        rc = parallel.Client(profile=self.profile) 
        lview = rc.load_balanced_view() 
        lview.retries = 10
        
        number_of_jobs = len(commands)
        self.logger.debug(number_of_jobs)

        tic = time.time()
        ar = lview.map(execute_command, commands)
        
        for i,r in enumerate(ar):
            self.logger.debug("task: %i finished on %s, %.3f percent finished at time %.3f "%(
                               i, r['host'], 100*((i+1)/float(number_of_jobs)), time.time()-tic ))

        self.logger.debug('done')









