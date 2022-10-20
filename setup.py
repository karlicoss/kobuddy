# see https://github.com/karlicoss/pymplate for up-to-date reference


from setuptools import setup, find_packages # type: ignore


def main() -> None:
    pkgs = find_packages('src')
    [pkg] = pkgs
    setup(
        name=pkg,
        use_scm_version={
            'version_scheme': 'python-simplified-semver',
            'local_scheme': 'dirty-tag',
        },
        setup_requires=['setuptools_scm'],

        zip_safe=False,

        packages=[pkg],
        package_dir={'': 'src'},
        package_data={pkg: ['py.typed']},

        ## ^^^ this should be mostly automatic and not requiring any changes

        url='https://github.com/karlicoss/kobuddy',
        author='Dima Gerasimov',
        author_email='karlicoss@gmail.com',
        description='Backup and extract data from your Kobo reader',

        install_requires=['pytz', 'dataset'],
        extras_require={
            'testing': ['pytest'],
            'linting': ['pytest', 'mypy', 'lxml', 'types-pytz'],
        },
        entry_points={'console_scripts': ['kobuddy = kobuddy.__main__:main']},
    )


if __name__ == '__main__':
    main()

# TODO
# from setuptools_scm import get_version
# https://github.com/pypa/setuptools_scm#default-versioning-scheme
# get_version(version_scheme='python-simplified-semver', local_scheme='no-local-version')
