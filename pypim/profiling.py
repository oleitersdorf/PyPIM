import pypim.driver as driver

def resetMetrics():
    for optype in [driver.MicrooperationType.mask, driver.MicrooperationType.rw, driver.MicrooperationType.logic, driver.MicrooperationType.move]:
        driver.resetMetrics(optype)

def getMetrics():
    optypes = {
        'mask': driver.MicrooperationType.mask,
        'rw': driver.MicrooperationType.rw,
        'logic': driver.MicrooperationType.logic,
        'move': driver.MicrooperationType.move
    }
    metrics = {name: driver.getMetrics(optype) for (name, optype) in optypes.items()}
    metrics['total'] = sum(metrics.values(), driver.Metrics(0, 0))
    return metrics

def printMetrics(metrics=None):
    if metrics is None:
        metrics = getMetrics()
    print('Type\tLatency\tEnergy')
    print('----------------------')
    for (name, res) in metrics.items():
        print(f'{name}\t{res.latency}\t{res.energy}')
    print()

class Profiler:
    def __init__(self, name = None):
        self.name = name

    def __enter__(self) -> None:
        self.metrics = getMetrics()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        new_metrics = getMetrics()
        print()
        if self.name is not None:
            print(self.name)
        printMetrics({name: new_metrics[name] - old for (name, old) in self.metrics.items()})
