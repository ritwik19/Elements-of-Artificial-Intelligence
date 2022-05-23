'''test_choose_team.py
Usage: 
python3 test_choose_team.py user1-user2-user3-a1/part3
It is not always possible to get the best solution so we will check whether the solution is below a threshold in order to pass the test case.
For final grading we will be using more complex test cases.
'''
import pytest
import numpy as np
import assign
import time
import signal
time_ = 100

def handler(signum, frame):
	raise Exception("timeout")

def get_solution(test_file):
	results = [[]]
	try:
		for i in assign.solver(test_file):
			results.append([i['assigned-groups'],i['total-cost']])
	except Exception:
		return results
	return results

def check_names(test_file,result):
	names_ = [j for i in [i.split('-') for i in result[0]] for j in i]
	names = set(names_)
	with  open(test_file,'r') as f:
		original_names = set()
		for i in f.readlines():
			original_names.add(i.split()[0])
	return (original_names==names and len(names)==len(original_names))

def check_solution(test_file,result,threshold = float('inf')):
	assert len(result) != 0, "No solution yielded in {} seconds".format(str(time_))
	assert result[-1] >= 0, "Score cannot be negative" 
	assert check_names(test_file,result) == True, 'Everyone should be assigned to a team'
	assert type(result[1]) in (int,float), 'Cost should be of type int or float'
	assert result[1] <= threshold, 'The cost is incorrect, it could be better'

def test_case_1():
	signal.signal(signal.SIGALRM, handler)
	signal.alarm(time_)
	test_file = 'test1.txt'
	check_solution(test_file,get_solution(test_file)[-1],30) 

def test_case_2():
	signal.signal(signal.SIGALRM, handler)
	signal.alarm(time_)
	test_file = 'test2.txt'
	check_solution(test_file,get_solution(test_file)[-1],50) 

def test_case_3():
	signal.signal(signal.SIGALRM, handler)
	signal.alarm(time_)
	test_file = 'test3.txt'
	check_solution(test_file,get_solution(test_file)[-1])  ## there is no threshold for this case. 