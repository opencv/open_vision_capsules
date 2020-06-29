from vcap.backend import BaseBackend


class BaseTFBackend(BaseBackend):
    def close(self):
        super().close()
        self.session.close()
