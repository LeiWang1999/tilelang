multi_python_version=("3.8" "3.9" "3.10" "3.11" "3.12")
for python_version in "${multi_python_version[@]}"; do
    echo "Installing Python ${python_version}..."
    apt-get install -y python${python_version}
done

pip install -r requirements-build.txt

# if dist and build directories exist, remove them
if [ -d dist ]; then
    rm -r dist
fi

# Build source distribution (disabled for now)
# python setup.py sdist --formats=gztar,zip

# Build wheels for different Python versions
echo "Building wheels for multiple Python versions..."
tox -e py38,py39,py310,py311,py312

if [ $? -ne 0 ]; then
    echo "Error: Failed to build the wheels."
    exit 1
else
    echo "Wheels built successfully."
fi