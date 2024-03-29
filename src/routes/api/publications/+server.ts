export const prerender = true;
import { json } from '@sveltejs/kit';

export const GET = async ({ url }) => {
  // console.log(url);
  const pathname = url.pathname;
  return json({ pathname });
};

// export const GET = async ({ url }) => {
//   console.log(url);
//   const pathname = url.pathname;
//   const jsonFilePaths = import.meta.glob('/src/routes/api/publications/**/*.json');
//   const bibFiles = import.meta.glob('/src/routes/api/publications/**/*.bib');

//   const fetchBib = await fetch('/src/routes/api/publications/**/*.bib');
//   console.log(fetchBib);

//   // remove template folder with pattern matching
//   delete jsonFilePaths['/src/routes/api/publications/template/metadata.json'];
//   delete bibFiles['/src/routes/api/publications/template/citation.bib'];

//   const jsonDataDict = await Promise.all(
//     Object.entries(jsonFilePaths).map(async ([path, resolver]) => {
//       // check if path contrains substring
//       if (!path.includes(pathname)) {
//         return ['', {}];
//       }
//       const { default: jsonData } = await resolver();
//       const jsonPath = path.slice(11, -14);
//       return [jsonPath, jsonData];
//     })
//   );

//   const jsonData = {};
//   jsonDataDict.forEach(([path, data]) => {
//     if (path == '') {
//       return;
//     }
//     jsonData[path] = data;
//   });

//   const bibDataDict = await Promise.all(
//     Object.entries(bibFiles).map(async ([path]) => {
//       if (!path.includes(pathname)) {
//         return ['', {}];
//       }
//       const { default: mdData } = await import(path + '?raw');
//       const mdPath = path.slice(11, -13);
//       return [mdPath, mdData];
//     })
//   );

//   const bibData = {};
//   bibDataDict.forEach(([path, data]) => {
//     if (path == '') {
//       return;
//     }
//     bibData[path] = data;
//   });

//   Object.entries(jsonData).forEach(([path, data]) => {
//     jsonData[path] = {
//       ...data,
//       bib: bibData[path]
//     };
//   });
//   // transform dict to list
//   const jsonDataList = Object.entries(jsonData).map(([path, data]) => {
//     return {
//       ...data,
//       path: path
//     };
//   });

//   return json(jsonDataList);
// };
