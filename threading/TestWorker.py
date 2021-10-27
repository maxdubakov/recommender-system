from threading import Thread


class DownloadWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            data = self.queue.get()
            try:
                pass
                # process(data)
            finally:
                self.queue.task_done()
