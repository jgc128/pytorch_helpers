from pycrayon import CrayonClient


def get_crayon_experiment(exp_name, hostname='127.0.0.1', overwrite=True):
    cc = CrayonClient(hostname=hostname)

    cc_exp = None

    experiments = cc.get_experiment_names()
    if exp_name in experiments:
        if overwrite:
            cc.remove_experiment(exp_name)
            cc_exp = cc.create_experiment(exp_name)
        else:
            cc_exp = cc.open_experiment(exp_name)
    else:
        cc_exp = cc.create_experiment(exp_name)

    return cc_exp
