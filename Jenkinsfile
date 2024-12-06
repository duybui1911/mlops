pipeline {
    agent any

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        registry = 'quandvrobusto/house-price-prediction-api'
        registryCredential = 'dockerhub'   
        GOOGLE_APPLICATION_CREDENTIALS = credentials('google-application-credentials')
        PINECONE_APIKEY = credentials('pinecone-apikey')   
    }

    stages {
        stage('Test') {
            agent {
                docker {
                    image 'python:3.9' 
                }
            }
            steps {
                echo 'Testing model correctness..'
                sh 'pip install -r requirements.txt && pytest'
            }
        }
        stage('Build') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    def imageName = "${registry}:v1.${BUILD_NUMBER}"
                    dockerImage = docker.build(imageName, "--file Dockerfile-app .")
                    echo 'Pushing image to dockerhub..'
                    docker.withRegistry('', registryCredential ) {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }
        stage('Deploy') {
            agent {
                kubernetes {
                    containerTemplate {
                        name 'helm' // Name of the container to be used for helm upgrade
                        image 'duong05102002/jenkins-k8s:latest' // The image containing helm
                    }
                }
            }
            steps {
                script {
                    container('helm') {
                        sh("helm upgrade --install app --set image.repository=${registry} \
                        --set image.tag=v1.${BUILD_NUMBER} ./helm_charts/app --namespace model-serving")
                    }
                }
            }
        }
    }
}