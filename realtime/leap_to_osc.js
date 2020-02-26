const Leap = require('leapjs');
const {Client} = require('node-osc');

const controller = new Leap.Controller({enabledGestures: true});
const client = new Client('127.0.0.1', 3334);

controller.loop(function(frame) {
  for (let i in frame.handsMap) {
    let hand = frame.handsMap[i];
    let pos = hand.palmPosition;
    client.send('/leap/' + hand.type + '/x', pos[0] / 100);
    client.send('/leap/' + hand.type + '/y', pos[1] / 100 - 2);
    client.send('/leap/' + hand.type + '/z', pos[2] / 100);
  }
});