import neptune.new as neptune


class NeptuneLogger:

    def __init__(self, project: str, api_token: str, tags: list,
                 parameters: dict, **kwargs) -> None:
        self.project = project
        self.api_token = api_token
        self.tags = tags
        self.parameters = parameters
        self.run = neptune.init(project=project, api_token=api_token,
                                source_files=[])
        self.run['sys/tags'].add(tags)
        self.run['parameters'] = parameters

    def log_metrics(self, metrics: dict) -> None:
        """
        :param metrics:
        :return:
        """
        for key, values in metrics.items():
            for val in values:
                self.run[key].log(val)

    def log_artifacts(self, artifacts):
        """

        :param artifacts:
        :return:
        """
        self.run['files'].upload_files(artifacts)
