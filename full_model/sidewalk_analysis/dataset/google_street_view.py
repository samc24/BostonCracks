import streetview
import google_streetview.api
import urllib


def get_image(lat, long, _id):
    panoids = streetview.panoids(lat=lat, lon=long)
    panoid = panoids[0].get('panoid')

    params = [{
      'size': '2048x1024',
      'location': '47.5542135, -122.33387',
      'heading': '151.78',
      'pitch': '-0.76',
      'key': 'AIzaSyDFbJlBANBIFfmyRXBDQuPfflGkcBpbCiw',
      'pano_id': panoid,
    }]

    results = google_streetview.api.results(params)
    link = results.links[0]
    name = "gsv_" + str(_id) + ".jpg"
    urllib.urlretrieve(link, "gsv_" + str(_id) + ".jpg")
    results.download_links('gsv_images/images')


get_image('-33.856938', '151.214489')
