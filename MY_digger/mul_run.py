
yellow = '\033[1;33m'
yellow_ = '\033[0m'


def run_mytest():
    import mul_mytest as mm
    import myplot
    import time

    # Reference.check()
    time.sleep(5)
    P = mm.RUN()
    P.results()
    myplot.show()


if __name__ == '__main__':
    # choicepcon = run_mychoice()[2]
    run_mytest()
