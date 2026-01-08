// ===============================
// Simple Harmonic Motion (Spring-Mass)
// Processing Java Mode
// ===============================

float A = 120;        // amplitude
float omega = 0.03;   // angular frequency
float t = 0;

float wallX = 80;     // fixed wall position
float equilibriumX = 360;  // equilibrium position
float blockSize = 40;

void setup() {
  size(900, 300);
  smooth();
}

void draw() {
  background(200);

  // time update
  t += 1;

  // SHM equation
  float x = equilibriumX + A * cos(omega * t);

  // draw wall
  drawWall(wallX);

  // draw spring
  drawSpring(wallX, x);

  // draw mass block
  drawBlock(x);

  // display info
  fill(0);
  textSize(14);
  text("Simple Harmonic Motion", 20, 25);
  text("x(t) = A cos(ωt)", 20, 45);
}

void drawWall(float x) {
  stroke(255);
  strokeWeight(4);
  line(x, 60, x, height - 60);
}

void drawSpring(float x1, float x2) {
  int coils = 18;
  float y = height / 2;
  float len = x2 - x1 - blockSize / 2;
  float step = len / coils;
  float amp = 18;

  stroke(255);
  strokeWeight(2);
  noFill();

  beginShape();
  vertex(x1, y);

  for (int i = 1; i < coils; i++) {
    float px = x1 + i * step;
    float py = y + ((i % 2 == 0) ? -amp : amp);
    vertex(px, py);
  }

  vertex(x2 - blockSize / 2, y);
  endShape();
}

void drawBlock(float x) {
  fill(0, 0, 255);
  noStroke();
  rectMode(CENTER);
  rect(x, height / 2, blockSize, blockSize);
}
