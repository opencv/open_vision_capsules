from vcap.backend import BaseBackend


class BaseTFBackend(BaseBackend):
    def close(self):
        self.session.close()
