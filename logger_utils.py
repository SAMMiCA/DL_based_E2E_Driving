def create_file(workspace, file_name):
    fname = workspace + file_name + ".csv"
    file = open(fname,"w")
    file.write("==========This is the beginning==========\n")
    file.write("File format is :\n")
    file.write("Steps,Episode,Reward\n")
    file.close()

# Appending data to the created file over times
def write_to_file(workspace,file_name,env_steps,episode,score):
    # Open file and append data
    fname = workspace + file_name + ".csv"
    file = open(fname,"a+")
    file.write(str(env_steps) + "," + str(episode) + "," + str(score)+ "\n")
    file.close()
    return 1


def create_local_eval_file(workspace,id):
    fname = workspace + "eval"+str(id) +".csv"
    file = open(fname,"w")
    file.write("==========This is the beginning==========\n")
    file.write("File format is :\n")
    file.write("Steps,Local_steps,Reward,Alpha\n")
    file.close()

def write_local_eval_to_file(workspace,env_steps,episode,score, id):
    # Open file and append data
    fname = workspace + "eval"+str(id) +".csv"
    file = open(fname,"a+")
    file.write(str(env_steps) + "," + str(episode) + "," + str(score)+ "\n")
    file.close()
    return 1
