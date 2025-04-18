pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage("Cloning github repo to Jenkins"){
            steps{
                script {
                    echo 'Cloning github repo to Jenkins................'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/benfls/MLOPS-COURSE-PROJECT-1.git']])
                }
            }
        }

        stage("Steting up our Virtual Environment and Installing dependencies"){
            steps{
                script {
                    echo 'Steting up our Virtual Environment and Installing dependencies................'
                    sh ''' 
                        python3 -m venv ${VENV_DIR}
                        . $VENV_DIR/bin/activate
                        pip install --upgrade pip
                        pip install -e .
                    '''
                }
            }
        }
    }
}

