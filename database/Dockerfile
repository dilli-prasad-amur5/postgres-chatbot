FROM postgres:15

# Install pgvector extension
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-15 \
    gcc \
    make \
    libc6-dev \
    git \
    && git clone https://github.com/pgvector/pgvector.git \
    && cd pgvector && make && make install \
    && apt-get remove -y gcc make libc6-dev git && apt-get autoremove -y && apt-get clean

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/

# Expose PostgreSQL default port
EXPOSE 5432
