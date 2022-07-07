import matlab.engine
eng = matlab.engine.start_matlab()

eng.run_HFO_detector(nargout=0)
eng.quit()