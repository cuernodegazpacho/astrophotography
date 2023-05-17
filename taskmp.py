# task executed in a worker process
def task():
    # report a message
    print(f'Task executing', flush=True)
    # block for a moment
    sleep(1)
    # report a message
    print(f'Task done', flush=True)
 
