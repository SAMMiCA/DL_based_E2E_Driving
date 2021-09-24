import json

class ParamsManager(object):
    def __init__(self, params_file):
        self.params = json.load(open(params_file,'r'))

    def get_params(self):
        return self.params

    def get_env_params(self):
        return self.params['env']

    def get_agent_params(self):
        return self.params['agent']

    def update_agent_params(self, **kwargs):
        """
        Update the hyper-parameters (and configuration parameters) used by the agent
        :param kwargs:  Comma-separated, hyper-parameter-key=value pairs. Eg.: lr=0.005, gamma=0.98
        :return: None
        """
        for key, value in kwargs.items():
            if key in self.params['agent'].keys():
                self.params['agent'][key] = value
    
    def update_env_params(self, **kwargs):
        """
        Update the hyper-parameters (and configuration parameters) used by the agent
        :param kwargs:  Comma-separated, hyper-parameter-key=value pairs. Eg.: lr=0.005, gamma=0.98
        :return: None
        """
        for key, value in kwargs.items():
            if key in self.params['env'].keys():
                self.params['env'][key] = value

    def export_json(self,params_file,json_data):
        with open(params_file, 'w') as f:
            json.dump(self.params, f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write("\n")



