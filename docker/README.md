>> docker-compose up -d
neu Dockerfile va docker-compose nam cung 1 noi thi 
>> build: .
neu dat khac cho thi
>> 
context: ./
dockerfile: docker/Dockerfile


>> PM2 (dustin-product-vimusic-server)

Start an app using all CPUs available + set a name :
    $ pm2 start app.js -i 0 --name "api"

    Restart the previous app launched, by name :
    $ pm2 restart api

    Stop the app :
    $ pm2 stop api

    Restart the app that is stopped :
    $ pm2 restart api

    Remove the app from the process list :
    $ pm2 delete api

    pm2 logs api