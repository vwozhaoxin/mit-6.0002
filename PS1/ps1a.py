###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # TODO: Your code here
    path ='./'
    new_dict ={}
    with open(path+filename,'r') as f:
        loop = True
        for line in f.readlines():
            flist = line.strip().split(',')
            new_dict[flist[0]] = flist[1]
    return new_dict
cow1 = load_cows('ps1_cow_data.txt')
cow2 = load_cows('ps1_cow_data_2.txt')
#print(cow1)
#print(cow2)
#print(load_cows('ps1_cow_data.txt'))
#print(load_cows('ps1_cow_data_2.txt'))
# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    t1= time.time()
    # TODO: Your code here
    results =[]
    sortlist = list(cows.values())
    newcows = cows.copy()
    sortlist.sort(reverse=True)
#    print(newcows)
#    print(sortlist)
    while len(sortlist)!=0:
        newlimit =limit
        result =[]
        n=0
        while int(min(sortlist)) <= newlimit: #最小值小于limit  可以添加新的cow           
            popvalues = int(sortlist[n])        
            if popvalues <= newlimit:    #小于limit 的最大值
                popkey = list(newcows.keys())[list(newcows.values()).index(str(popvalues))]# 找到对应的cow
                result.append(popkey)    #添加到trip
                newlimit -= popvalues    #limit更新
                newcows.pop(popkey)      #更新cow  一定要根本性 因为查找key的时候需要
#                print(popvalues,sortlist)
                sortlist.remove(str(popvalues)) #列表中删除对应的value
            else:
                n+=1
            if len(sortlist)==0:
                break
        results.extend([result])
    t2=time.time()
#    print('greedy cost time:', (t2-t1))
    return results
#print(greedy_cow_transport(cow1,10))

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    start = time.time()
    brutelist = [trip for trip in get_partitions(cows.keys())]
    bestnumber =100000
    for outloop in brutelist:
        results =[]
        
        for middleloop in outloop:
            result=[]
            listlimit=0
            if type(middleloop)==list:
                for innerloop in middleloop:
                    listlimit+=int(cows[innerloop])
                if listlimit<=limit:
                    result.append(middleloop)
                else:
                    break
            else:
                result.appen(middleloop)
            results.append(result)
        if result!=[]:
            number = len(results)
            if number<bestnumber:
               bestnumber = number
               best_results = results
#               print(best_results)
    end = time.time()
#    print('brute cost time:', end-start)
    return best_results    
#print(brute_force_cow_transport(cow1))                        
        
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # TODO: Your code here
    start1=time.time()
    result1= greedy_cow_transport(cow1)
    end1=time.time()
    start2=time.time()
    result2= brute_force_cow_transport(cow1)
    end2=time.time()
    
    print('len of greedy ',len(result1),'cost time:', end1-start1)
    print('len of brute ',len(result2),'cost time:', end2-start2)

        
compare_cow_transport_algorithms()