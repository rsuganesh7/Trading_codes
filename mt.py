import random
import time
import threading

numbers = []
random_int = 0
def rand_num():
    global random_int
    while True:
        random_int = random.randint(1,1
                                    000)
        time.sleep(0.5)
        

def condition():
    global numbers
    while len(numbers) <10:
        if random_int > 100 and random_int %5 ==0 and random_int not in numbers :
            numbers.append(random_int)
            
    event.set()
            
            
        
def display_numbers():
    """Display the generated numbers indefinitely."""
    while True:
        generated_numbers = len(numbers)
        print(f"{generated_numbers} numbers generated: {numbers}")
        time.sleep(3)

event = threading.Event()
stream_thread = threading.Thread(target= rand_num)
stream_thread.start()

condition_thread = threading.Thread(target= condition,daemon=True)
condition_thread.start()

display_thread = threading.Thread(target= display_numbers)
display_thread.start()