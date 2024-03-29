export async function load({ fetch }) {
  const publications = await import('./publications/publications.md');
  const bioResp = await fetch('/api/bio');
  const bio = await bioResp.json();

  const newsResp = await fetch('/api/news');
  const news = await newsResp.json();

  const interestsResp = await fetch('/api/interests');
  const interests = await interestsResp.json();

  return {
    publications,
    bio,
    news,
    interests
  };
}
