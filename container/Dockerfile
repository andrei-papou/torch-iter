FROM docker.io/library/python:3.10

RUN apt update && apt install -y \
    git \
    curl

RUN pip install -U pip
COPY requirements requirements
RUN pip install -r requirements/local.txt

# Install starship shell
RUN curl -sS https://starship.rs/install.sh > install_starship.sh && \
    chmod +x install_starship.sh && \
    ./install_starship.sh --yes \
    && echo "eval \"\$(starship init bash)\"" >> ~/.bashrc

