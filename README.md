Kobuddy is a tool to access Kobo Reader sqlite database and extract useful things from it.

It's a bit adhoc at the moment, you're gonna need [kython](https://github.com/karlicoss/kython), [export-kobo](https://github.com/pettarin/export-kobo) and few other pypi dependencies.

Some paths are hardcoded in `__init__.py` but should be fairly easy to work around it.

At the moment I'm working on converting it in a proper Python package so it doesn't have any random dependencies.
