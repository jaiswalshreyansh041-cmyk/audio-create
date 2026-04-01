import fs from 'fs';
import path from 'path';

function findFile(dir, target) {
  try {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const fullPath = path.join(dir, file);
      try {
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory()) {
          findFile(fullPath, target);
        } else if (file === target) {
          console.log('Found:', fullPath);
        }
      } catch (e) {}
    }
  } catch (e) {}
}

findFile('/workspace', 'overview.txt');
findFile('/home', 'overview.txt');
findFile('/app', 'overview.txt');
console.log('Done');
