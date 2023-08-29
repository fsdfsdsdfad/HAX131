



from concurrent import futures
import logging
import ai_pytorch as ai
import flask
from flask import request

app = flask.Flask(__name__)

"""
class Greeter(buf_pb2_grpc.SolverServicer):
    def Hi(self, request, context):
        return buf_pb2.HiResponse(response='hi %s' % request.name)

    def Solve(self, request, context):
        imgs = request.img

        predictions = ai.predict(imgs[0], request.text)

        return buf_pb2.SolveResponse(answers=predictions)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    buf_pb2_grpc.add_SolverServicerServicer_to_server(Greeter(), server)
    server.add_insecure_port('0.0.0.0:8888')
    server.start()
    print("serving on 0.0.0.0:8888")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
"""



import json
@app.route('/solve', methods = ['POST', 'GET'])
async def solve():
    data = request.json
    new_keys = {int(k): v for k, v in data["images_map"].items()}
    data["images_map"] = new_keys

    # while len(data["images_map"]) < 9:
        # data["images_map"].append(data["images_map"][0])

    predictions = ai.predict(data["label"], data["images_map"])
    # <string, bool> dict
    # return as json

    response = {
  "errorId": 0,
  "errorCode": "",
  "status": "ready",
  "solution": {
    "objects": predictions,
  },
  "taskId": "5aa8be0c-94a5-11ec-80d7-00163f00a53c"
}

    return app.response_class(
        response=json.dumps(response),
        status=200,
        mimetype='application/json'
    )



if __name__ == '__main__':
 app.run(host='0.0.0.0', port=8082)






