from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gmol-core")
except PackageNotFoundError:
    pass

del PackageNotFoundError, version
