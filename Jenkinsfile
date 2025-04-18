pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "dynamic-music-456811-m8"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
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

        stage("Setting up our Virtual Environment and Installing dependencies"){
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

        stage("Building and push docker image to GCR"){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script {
                        echo 'Building and push docker image to GCR................'
                        sh ''' 
                            export PATH=$PATH:${GCLOUD_PATH}

                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                            gcloud config set project ${GCP_PROJECT}

                            gcloud auth configure-docker --quiet

                            docker builds -t gcr.io/${GCP_PROJECT}/mlops-project:latest .

                            docker push gcr.io/${GCP_PROJECT}/mlops-project:latest
                             
                        '''
                    }
                }
            }
        }
    }
}

