class VerificationWorkflow:
    def __init__(self, model, graph, labels, fingerprinting_args):
        self.fingerprinter = Fingerprinting(model, graph, labels, **fingerprinting_args)
        self.fingerprints = None

    def offline_phase(self):
        # 1. Generate fingerprints and record expected outputs
        self.fingerprints = self.fingerprinter.select_fingerprints()

    def online_phase(self, queried_model):
        # 2. Query fingerprint nodes, compare predictions
        # 3. Return detection result (True if any mismatch)
        pass
