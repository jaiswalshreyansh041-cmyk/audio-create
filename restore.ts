import { execSync } from 'child_process';
try {
  execSync('git checkout app.py fix_indent.py requirements.txt');
  console.log('Files restored successfully.');
} catch (e) {
  console.error('Failed to restore files:', e);
}
